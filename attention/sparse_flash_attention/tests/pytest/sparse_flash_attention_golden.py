#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
import os
import torch
import copy

import generate_tensor_data


def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        print(
            f"[ERROR]concat_tensor: 相加的两个tensor 维数不同! shape1 = {shape1}, shape2 =  {shape2}"
        )
        return None, None
    if len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            print(
                f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}"
            )
            return None, None
        if tnd_flag:
            concatenated_tensor = torch.cat((tensor1, tensor2), dim=2)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
        else:
            d1 = int(shape1[2] / n)
            d2 = int(shape2[2] / n)
            tensor1_bsnd = tensor1.reshape(shape1[0], shape1[1], n, d1)
            tensor2_bsnd = tensor2.reshape(shape2[0], shape2[1], n, d2)
            concatenated_tensor = torch.cat((tensor1_bsnd, tensor2_bsnd), dim=3)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
            concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 4:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            print(
                f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}"
            )
            return None, None
        concatenated_tensor = torch.cat((tensor1, tensor2), dim=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape


def preprocessing(fa_param):
    numHeads = fa_param["numHeads"]
    numKeyValueHeads = fa_param["numKeyValueHeads"]

    def _clone(t):
        if torch.is_tensor(t):
            return t.clone()
        return copy.deepcopy(t)

    fa_param["q_tensor_old"] = _clone(fa_param["q_tensor"])
    fa_param["q_shape_old"] = copy.deepcopy(fa_param["q_shape"])
    fa_param["k_tensor_old"] = _clone(fa_param["k_tensor"])
    fa_param["k_shape_old"] = copy.deepcopy(fa_param["k_shape"])

    q_new_tensor, q_new_shape = concat_tensor(
        fa_param["q_tensor"],
        fa_param["q_shape"],
        fa_param["q_rope_tensor"],
        fa_param["q_rope_shape"],
        numHeads,
        fa_param["tnd_flag"],
    )
    if q_new_tensor is not None:
        fa_param["q_tensor"] = q_new_tensor
        fa_param["q_shape"] = q_new_shape
    else:
        print("[ERROR]q tensor的预处理异常，输出空tensor！")

    k_new_tensor, k_new_shape = concat_tensor(
        fa_param["k_tensor"],
        fa_param["k_shape"],
        fa_param["k_rope_tensor"],
        fa_param["k_rope_shape"],
        numKeyValueHeads,
        fa_param["kv_tnd_flag"],
    )
    if k_new_tensor is not None:
        fa_param["k_tensor"] = k_new_tensor
        fa_param["k_shape"] = k_new_shape
    else:
        print("[ERROR]k tensor的预处理异常，输出空tensor！")

    return fa_param


def _n_trans_shape_to_bnsd(
    tensor, shape, layout, headnums=None, act_seq=None, tensor_name=None
):
    t = tensor

    if layout in ["BSH", "BSH_NBSD"]:
        if headnums is None:
            return t, shape
        B = shape[0]
        S = shape[1]
        if tensor_name != "sparseIndices":
            H = shape[2]
            N = headnums
            D = H // N
            out = t.reshape(B, S, N, D).permute(0, 2, 1, 3).contiguous()
            return out, [B, N, S, D]
        else:
            out = t.permute(0, 2, 1, 3).contiguous()
            return out, [B, headnums, S, shape[3]]
    elif layout in ["BSND", "PA_BSND"]:
        B = shape[0]
        S = shape[1]
        N = shape[2]
        D = shape[3]
        out = t.permute(0, 2, 1, 3).contiguous()
        return out, [B, N, S, D]
    elif layout in ["TND", "TND_NTD"]:
        T = shape[0]
        N = shape[1]
        D = shape[2]
        B = len(act_seq)
        S = max(act_seq)
        new_tensor = torch.zeros((B, N, S, D), dtype=t.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = t[t_start:t_end, n_index, :]
            t_start += act_s
        return new_tensor, [B, N, S, D]
    else:
        return t, shape


def trans_tnd_actseq(list):
    list_len = len(list)
    list_new = []
    list_new.append(list[0])
    for i in range(list_len - 1):
        new_item = list[i + 1] - list[i]
        if new_item >= 0:
            list_new.append(new_item)
        else:
            print(
                f"[ERROR]trans_tnd_actseq: Wrong input actseq:{list}, in loop {i}, item {new_item} < 0"
            )
    print(f"[INFO]before trans: {list}")
    print(f"[INFO]after trans: {list_new}")
    return list_new


def trans_bnsd_to_layout(tensor, shape, layout, act_q=None):
    if layout == "BSH":
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout in ["BSND", "PA_BSND"]:
        output = tensor.permute(0, 2, 1, 3).contiguous()
        return output
    elif layout in ["BSND_NBSD", "BNSD_NBSD", "BSH_NBSD"]:
        output = tensor.permute(1, 0, 2, 3).contiguous()
        return output
    elif layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        B = tensor.shape[0]
        N = tensor.shape[1]
        D = tensor.shape[3]
        output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s

            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :act_s, :]
            t_start += act_s
        if layout == "TND_NTD":
            output = output.permute(1, 0, 2).contiguous()
        return output
    else:
        return tensor


def trans_input_dtype(input_dtype):
    if input_dtype == "fp16":
        return "float16"
    elif input_dtype == "int8":
        return "int8"
    elif input_dtype == "uint8":
        return "uint8"
    elif input_dtype == "bf16":
        return "bfloat16"
    elif input_dtype == "bool":
        return "bool"
    elif input_dtype == "int32":
        return "int32"
    elif input_dtype == "fp32":
        return "float32"
    elif input_dtype == "int4":
        return "int4"
    else:
        return input_dtype


def get_param_fus(input_tensor_dict, params):
    fa_param = {}
    fa_param["normal_flag"] = True
    fa_param["actualSeqLengths_q_raw"] = params["actualseqlengths"]
    fa_param["actualSeqLengths_raw"] = params["actualseqlengthskv"]
    fa_param["scaleValue"] = params["scalevalue"]
    fa_param["layout_query"] = params["layout_query"]
    fa_param["layout_kv"] = params["layout_kv"]
    fa_param["sparse_mode"] = params["sparsemode"]
    fa_param["sparse_blocksize"] = params["sparse_blocksize"]

    fa_param["q_shape"] = params["shape_input"]["query"]
    fa_param["q_dtype"] = trans_input_dtype(params["dtype_input"]["query"])
    fa_param["q_tensor"] = input_tensor_dict["query"]

    fa_param["k_shape"] = params["shape_input"]["key"]
    fa_param["k_dtype"] = trans_input_dtype(params["dtype_input"]["key"])
    fa_param["k_tensor"] = input_tensor_dict["key"]

    fa_param["v_shape"] = fa_param["k_shape"]
    fa_param["v_dtype"] = fa_param["k_dtype"]
    fa_param["v_tensor"] = fa_param["k_tensor"]

    fa_param["sparse_indices_shape"] = params["shape_input"]["sparse_indices"]
    fa_param["sparse_indices_dtype"] = trans_input_dtype(
        params["dtype_input"]["sparse_indices"]
    )
    fa_param["sparse_indices_tensor"] = input_tensor_dict["sparse_indices"]

    fa_param["bt_shape"] = params["shape_input"]["block_table"]
    fa_param["bt_dtype"] = trans_input_dtype(params["dtype_input"]["block_table"])
    fa_param["block_table"] = input_tensor_dict["block_table"]

    fa_param["q_rope_shape"] = params["shape_input"]["query_rope"]
    fa_param["q_rope_dtype"] = trans_input_dtype(params["dtype_input"]["query_rope"])
    fa_param["q_rope_tensor"] = input_tensor_dict["query_rope"]

    fa_param["k_rope_shape"] = params["shape_input"]["key_rope"]
    fa_param["k_rope_dtype"] = trans_input_dtype(params["dtype_input"]["key_rope"])
    fa_param["k_rope_tensor"] = input_tensor_dict["key_rope"]

    fa_param["sinks_shape"] = params["shape_input"].get("sinks")
    fa_param["sinks_dtype"] = trans_input_dtype(
        params["dtype_input"].get("sinks", "fp32")
    )
    fa_param["sinks_tensor"] = input_tensor_dict.get("sinks")

    fa_param["out_shape"] = fa_param["q_shape"]
    fa_param["out_dtype"] = trans_input_dtype(params["dtype_output"][0])
    fa_param["softmax_max_shape"] = params["shape_output"]["softmax_max"]
    fa_param["softmax_sum_shape"] = params["shape_output"]["softmax_sum"]
    fa_param["return_softmax_lse"] = params["return_softmax_lse"]

    if fa_param["layout_query"] in ["TND"]:
        fa_param["b"] = len(fa_param["actualSeqLengths_q_raw"])
        fa_param["numHeads"] = fa_param["q_shape"][1]
    else:
        fa_param["b"] = fa_param["q_shape"][0]
        fa_param["numHeads"] = fa_param["q_shape"][2]

    fa_param["sparse_blockcount"] = fa_param["sparse_indices_shape"][-1]

    fa_param["blocksize"] = 1
    fa_param["numKeyValueHeads"] = fa_param["k_shape"][-2]
    if fa_param["layout_kv"] in ["PA_BSND"]:
        fa_param["blocksize"] = params["block_size"]

    fa_param["pa_flag"] = True if fa_param["layout_kv"] in ["PA_BSND"] else False
    fa_param["tnd_flag"] = True if fa_param["layout_query"] in ["TND"] else False
    fa_param["kv_tnd_flag"] = True if fa_param["layout_kv"] in ["TND"] else False

    if 0 in fa_param["q_shape"]:
        fa_param["normal_flag"] = False
        print("[WARNING]:异常-->  q为空场景！")
    if len(fa_param["k_shape"]) == 0 or 0 in fa_param["k_shape"]:
        fa_param["normal_flag"] = False
        print("[WARNING]:异常-->  k为空场景！")
    if len(fa_param["v_shape"]) == 0 or 0 in fa_param["v_shape"]:
        fa_param["normal_flag"] = False
        print("[WARNING]:异常-->  v为空场景！")

    return fa_param


def generate_input_tensors(params):
    tensor_keys = [
        "query",
        "key",
        "sparse_indices",
        "block_table",
        "query_rope",
        "key_rope",
    ]
    raw = {}
    for key in tensor_keys:
        raw[key] = generate_tensor_data.gen_tensor_data(params, key)

    sinks = None
    if params.get("use_sinks", False):
        sinks = generate_tensor_data.gen_tensor_data(params, "sinks")

    result = {
        "query": raw["query"],
        "key": raw["key"],
        "value": raw["key"],
        "sparse_indices": raw["sparse_indices"],
        "block_table": raw["block_table"],
        "query_rope": raw["query_rope"],
        "key_rope": raw["key_rope"],
        "sinks": sinks,
        "scale_value": params["scalevalue"],
        "sparse_block_size": params["sparse_blocksize"],
        "layout_query": params["layout_query"],
        "layout_kv": params["layout_kv"],
        "sparse_mode": params["sparsemode"],
        "attention_mode": params["attention_mode"],
        "pre_tokens": (1 << 63) - 1,
        "next_tokens": (1 << 63) - 1,
        "return_softmax_lse": params.get("return_softmax_lse", False),
    }

    return result


def compute_cpu(input_tensor_dict, params, return_bench=False):
    try:
        out_data = compute_golden(input_tensor_dict, params, return_bench=return_bench)
        return out_data, input_tensor_dict, params
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[ERROR] compute_cpu failed: {e}")
        return None, input_tensor_dict, params


def _prepare_fa_param(input_tensor_dict, params):
    fa_param = get_param_fus(input_tensor_dict, params)
    if not fa_param["normal_flag"]:
        return fa_param

    actualSeqLengths_q = fa_param["actualSeqLengths_q_raw"]
    actualSeqLengths_kv = fa_param["actualSeqLengths_raw"]
    layoutQuery = fa_param["layout_query"]
    numHeads = fa_param["numHeads"]
    numKeyValueHeads = fa_param["numKeyValueHeads"]

    fa_param = preprocessing(fa_param)

    if fa_param["tnd_flag"]:
        fa_param["actualSeqLengths_q"] = trans_tnd_actseq(actualSeqLengths_q)
        if fa_param["pa_flag"]:
            fa_param["actualSeqLengths_kv"] = actualSeqLengths_kv
        q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param["q_tensor"],
            fa_param["q_shape"],
            layoutQuery,
            numHeads,
            fa_param["actualSeqLengths_q"],
        )
    else:
        q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param["q_tensor"],
            fa_param["q_shape"],
            layoutQuery,
            numHeads,
            actualSeqLengths_q,
        )
    if fa_param["kv_tnd_flag"]:
        fa_param["actualSeqLengths_kv"] = trans_tnd_actseq(actualSeqLengths_kv)
        k_bnsd_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param["k_tensor"],
            fa_param["k_shape"],
            fa_param["layout_kv"],
            numKeyValueHeads,
            fa_param["actualSeqLengths_kv"],
        )
        v_bnsd_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param["v_tensor"],
            fa_param["v_shape"],
            fa_param["layout_kv"],
            numKeyValueHeads,
            fa_param["actualSeqLengths_kv"],
        )
    else:
        k_bnsd_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param["k_tensor"],
            fa_param["k_shape"],
            fa_param["layout_kv"],
            numKeyValueHeads,
            actualSeqLengths_kv,
        )
        v_bnsd_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param["v_tensor"],
            fa_param["v_shape"],
            fa_param["layout_kv"],
            numKeyValueHeads,
            actualSeqLengths_kv,
        )
    if fa_param["tnd_flag"]:
        sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param["sparse_indices_tensor"],
            fa_param["sparse_indices_shape"],
            layoutQuery,
            numKeyValueHeads,
            fa_param["actualSeqLengths_q"],
            "sparseIndices",
        )
    else:
        sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param["sparse_indices_tensor"],
            fa_param["sparse_indices_shape"],
            layoutQuery,
            numKeyValueHeads,
            actualSeqLengths_q,
            "sparseIndices",
        )

    fa_param["q_bnsd_tensor"] = q_bnsd_tensor
    fa_param["q_bnsd_shape"] = q_bnsd_shape
    fa_param["k_bnsd_tensor"] = k_bnsd_tensor
    fa_param["k_bnsd_shape"] = k_bnsd_shape
    fa_param["v_bnsd_tensor"] = v_bnsd_tensor
    fa_param["v_bnsd_shape"] = v_bnsd_shape
    fa_param["sparse_indices_bnsd_tensor"] = sparse_indices_bnsd_tensor
    fa_param["sparse_indices_bnsd_shape"] = sparse_indices_bnsd_shape
    fa_param["ks_max"] = 0

    if not fa_param["tnd_flag"]:
        if len(actualSeqLengths_q) == 0:
            fa_param["actualSeqLengths_q"] = [q_bnsd_shape[2]] * q_bnsd_shape[0]
        elif len(actualSeqLengths_q) == 1:
            fa_param["actualSeqLengths_q"] = [actualSeqLengths_q[0]] * q_bnsd_shape[0]
        else:
            fa_param["actualSeqLengths_q"] = actualSeqLengths_q
    if not fa_param["kv_tnd_flag"]:
        if len(actualSeqLengths_kv) == 0:
            fa_param["actualSeqLengths_kv"] = [k_bnsd_shape[2]] * k_bnsd_shape[0]
        elif len(actualSeqLengths_kv) == 1:
            fa_param["actualSeqLengths_kv"] = [actualSeqLengths_kv[0]] * k_bnsd_shape[0]
        else:
            fa_param["actualSeqLengths_kv"] = actualSeqLengths_kv

    return fa_param


def _generate_sparse_indices(
    input_tensor_dict, fa_param, params, out_shape, layoutQuery
):
    sparse_indices_bnsd_shape = fa_param["sparse_indices_bnsd_shape"]
    sparse_indices_tensor = torch.full(sparse_indices_bnsd_shape, -1, dtype=torch.int32)

    for b in range(sparse_indices_bnsd_shape[0]):
        act_seqlen_kv = fa_param["actualSeqLengths_kv"][b]
        act_seqlen_q = fa_param["actualSeqLengths_q"][b]

        for n in range(sparse_indices_bnsd_shape[1]):
            for s in range(sparse_indices_bnsd_shape[2]):
                if fa_param["sparse_mode"] == 0:
                    threshold = act_seqlen_kv
                elif fa_param["sparse_mode"] == 3:
                    threshold = act_seqlen_kv - act_seqlen_q + s + 1

                if threshold <= 0:
                    continue
                valid_blocks_max = math.ceil(
                    max(0, threshold) / fa_param["sparse_blocksize"]
                )
                block_indices = torch.randperm(valid_blocks_max - 1, dtype=torch.int32)
                valid_blocks_topk = min(valid_blocks_max, fa_param["sparse_blockcount"])
                sparse_indices_tensor[b, n, s, : valid_blocks_topk - 1] = block_indices[
                    : valid_blocks_topk - 1
                ]
                sparse_indices_tensor[b, n, s, valid_blocks_topk - 1] = (
                    valid_blocks_max - 1
                )

    if fa_param["tnd_flag"]:
        sparse_indices_tensor_tnd = trans_bnsd_to_layout(
            sparse_indices_tensor,
            out_shape,
            layoutQuery,
            fa_param["actualSeqLengths_q"],
        )
        input_tensor_dict["sparse_indices"] = sparse_indices_tensor_tnd.to(torch.int32)
    else:
        input_tensor_dict["sparse_indices"] = sparse_indices_tensor.permute(
            0, 2, 1, 3
        ).contiguous()

    fa_param["sparse_indices_bnsd_tensor"] = sparse_indices_tensor


def _generate_block_table_and_cache(
    input_tensor_dict, fa_param, params, actualSeqLengths_kv, out_shape
):
    if fa_param["pa_flag"]:
        blockSize = fa_param["blocksize"]
        blockTableShape = fa_param["bt_shape"]
        blockNum = params["block_num"]
        if 0 in fa_param["bt_shape"]:
            print("[WARNING]:block_table为空场景，输出空tensor！")
            return torch.zeros(out_shape)
        blockNumPerBlock = []
        block_num_min = 0
        for actual_seq in actualSeqLengths_kv:
            blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
            block_num_min += math.ceil(actual_seq / blockSize)
        if block_num_min > blockNum:
            print(
                f"[ERROR]Wrong input k_cache_shape: get blockNum = {blockNum}, but expect blockNum > {block_num_min}"
            )
            exit(1)
        block_idx_list = torch.randperm(blockNum, dtype=torch.int32)
        block_idx = 0
        block_table = torch.full(
            (blockTableShape[0], blockTableShape[1]), -1, dtype=torch.int32
        )
        block_table_batch_idx = 0
        for idx in blockNumPerBlock:
            for j in range(idx):
                block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
                block_idx += 1
            block_table_batch_idx += 1
        fa_param["block_table"] = block_table
        input_tensor_dict["block_table"] = block_table
        k_dtype = params.get("dtype_input", {}).get("key", "unknown")
        v_dtype = params.get("dtype_input", {}).get("value", "unknown")
        print(f"[PageAtten]Input Kdtype:{k_dtype} Vdtype:{v_dtype}")
        kv_pa_preprocessing(input_tensor_dict, fa_param, params)
    else:
        kv_nopa_preprocessing(input_tensor_dict, fa_param)


def compute_golden(input_tensor_dict, params, return_bench=False):
    print("cpu执行中...")

    def _to_cpu_float32_tensor(val):
        if torch.is_tensor(val):
            if val.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                return val.float().cpu()
            return val.cpu().float()
        return val

    cpu_input = {
        key: _to_cpu_float32_tensor(val) for key, val in input_tensor_dict.items()
    }
    fa_param = _prepare_fa_param(cpu_input, params)
    if not fa_param["normal_flag"]:
        return torch.zeros(fa_param["out_shape"])

    out_shape = fa_param["out_shape"]
    layoutQuery = fa_param["layout_query"]
    actualSeqLengths_kv = fa_param["actualSeqLengths_raw"]

    def _to_cpu_raw(val):
        if torch.is_tensor(val):
            return val.cpu()
        return val

    fa_param["_raw_key"] = _to_cpu_raw(input_tensor_dict.get("key"))
    fa_param["_raw_key_rope"] = _to_cpu_raw(input_tensor_dict.get("key_rope"))
    fa_param["_raw_query"] = _to_cpu_raw(input_tensor_dict.get("query"))
    fa_param["_raw_query_rope"] = _to_cpu_raw(input_tensor_dict.get("query_rope"))

    _generate_sparse_indices(
        input_tensor_dict, fa_param, params, out_shape, layoutQuery
    )
    _generate_block_table_and_cache(
        input_tensor_dict, fa_param, params, actualSeqLengths_kv, out_shape
    )

    y_all, softmax_max, softmax_sum = _t_increattention_bnsd(fa_param)
    y_all = trans_bnsd_to_layout(
        y_all, out_shape, layoutQuery, fa_param["actualSeqLengths_q"]
    )
    if not return_bench:
        return y_all, softmax_max, softmax_sum
    # 三方标杆(cross_check 用):同一份已生成的输入数据上,去掉 exp 量化重算一遍。
    # 必须在同一次 compute_golden 内完成——sparse_indices/block_table/kv_cache 由
    # 上面的随机生成步骤原地写入,外层重调会用新的随机值,三方就不再对齐同一份数据。
    print("[INFO]:cross_check 开启，计算高精度三方标杆...")
    y_bench, _, _ = _t_increattention_bnsd(fa_param, quantize_exp=False)
    y_bench = trans_bnsd_to_layout(
        y_bench, out_shape, layoutQuery, fa_param["actualSeqLengths_q"]
    )
    return y_all, softmax_max, softmax_sum, y_bench


def gatherKV(
    k_tensor,
    v_tensor,
    sparse_indices_Indices,
    sparse_blocksize,
    sparse_blockcount,
    batch,
    n2Idx,
    s1Idx,
    curr_actualSeq,
    curr_actualSeq_q,
    sparse_mode,
):
    s2_sparse = list()
    if sparse_mode == 0:
        threshold = curr_actualSeq
    elif sparse_mode == 3:
        delta_s = curr_actualSeq - curr_actualSeq_q
        threshold = delta_s + s1Idx + 1
    validCount = min(sparse_blockcount, math.ceil(threshold / sparse_blocksize))
    for i in range(validCount):
        sparseIndicesId = sparse_indices_Indices[i]

        if sparseIndicesId == -1:
            break
        begin_Idx = sparseIndicesId * sparse_blocksize
        end_Idx = (
            begin_Idx + sparse_blocksize
            if begin_Idx + sparse_blocksize <= curr_actualSeq
            else curr_actualSeq
        )
        if begin_Idx >= threshold:
            continue
        if end_Idx <= threshold:
            s2_sparse.extend(range(begin_Idx, end_Idx))
        else:
            s2_sparse.extend(range(begin_Idx, threshold))

    emptyFlag = False
    if len(s2_sparse) == 0:
        k_sparse, v_sparse = [], []
        emptyFlag = True
    else:
        k_sparse, v_sparse = (
            k_tensor[batch, n2Idx, s2_sparse, :],
            v_tensor[batch, n2Idx, s2_sparse, :],
        )

    return emptyFlag, k_sparse, v_sparse


def softmax(x, sinks=None):
    x = x.float()
    if sinks is not None:
        sinks = sinks.float().unsqueeze(1)
        x_max = torch.max(torch.cat([x, sinks], dim=-1), dim=-1, keepdim=True).values
    else:
        x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    if sinks is not None:
        x_sum = x_sum + torch.exp(sinks - x_max)
    ans = y / x_sum
    return ans, x_max, x_sum


def bmm1(q_curr, k_sparse, calc_dtype):
    # 极值输入（如 key_rope=BF16 上界）下 bmm1 真值可能超出 fp32 表示范围，
    # 此时采用 NPU matmul 结果
    # 其他场景保持 CPU matmul
    cpu_matmul = torch.matmul(q_curr.float(), k_sparse.float().T)
    if calc_dtype is not torch.bfloat16:
        return cpu_matmul
    try:
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            return cpu_matmul
        device = torch.device("npu")
        q_dev = q_curr.to(device=device, dtype=torch.bfloat16)
        k_dev = k_sparse.to(device=device, dtype=torch.bfloat16)
        npu_matmul = torch.matmul(q_dev, k_dev.T).float().cpu()
        keep = torch.isfinite(npu_matmul) & torch.isfinite(cpu_matmul)
        return torch.where(keep, cpu_matmul, npu_matmul)
    except Exception:
        return cpu_matmul


def _t_increattention_bnsd(fa_param, quantize_exp=True):
    batch_size = fa_param["b"]
    numheads = fa_param["numHeads"]
    numKeyValueHeads = fa_param["numKeyValueHeads"]
    actualSeqLengths_q = fa_param["actualSeqLengths_q"]
    actualSeqLengths_kv = fa_param["actualSeqLengths_kv"]
    scaleValue = fa_param["scaleValue"]
    sparse_blocksize = fa_param["sparse_blocksize"]
    sparse_blockcount = fa_param["sparse_blockcount"]
    sparseIndicesIndices = fa_param["sparse_indices_bnsd_tensor"]
    out_shape_bnsd = fa_param["q_bnsd_shape"]
    out_shape_bnsd[-1] = fa_param["v_bnsd_shape"][-1]
    sparse_mode = fa_param["sparse_mode"]
    g = numheads // numKeyValueHeads
    sinks_tensor = fa_param.get("sinks_tensor")
    if sinks_tensor is not None:
        sinks_tensor = sinks_tensor.cpu().float()

    q_bnsd_tensor = fa_param["q_bnsd_tensor"].float()
    k_bnsd_tensor = fa_param["k_bnsd_tensor"].float()
    v_bnsd_tensor = k_bnsd_tensor[..., :512].clone()

    matmul_dtype = torch.float32
    y = torch.zeros(out_shape_bnsd, dtype=torch.float32)
    softmax_max = torch.zeros(fa_param["softmax_max_shape"], dtype=torch.float32)
    softmax_sum = torch.zeros(fa_param["softmax_sum_shape"], dtype=torch.float32)

    def set_softmax_lse(
        batch_idx, n2_idx, s1_idx, q_last_prefill, max_value, sum_value
    ):
        if not fa_param["return_softmax_lse"]:
            return
        if fa_param["layout_query"] == "BSND":
            softmax_max[batch_idx, n2_idx, s1_idx, :] = max_value
            softmax_sum[batch_idx, n2_idx, s1_idx, :] = sum_value
        else:
            softmax_max[n2_idx, q_last_prefill + s1_idx, :] = max_value
            softmax_sum[n2_idx, q_last_prefill + s1_idx, :] = sum_value

    for batch in range(batch_size):
        curr_actualSeq_q = actualSeqLengths_q[batch]
        curr_actualSeq = actualSeqLengths_kv[batch]
        qLastPrefill = 0 if batch == 0 else sum(actualSeqLengths_q[0:batch])
        for n2Idx in range(numKeyValueHeads):
            for s1Idx in range(curr_actualSeq_q):
                if s1Idx < curr_actualSeq_q - curr_actualSeq and sparse_mode != 0:
                    y[batch, n2Idx * g : (n2Idx + 1) * g, s1Idx, :] = torch.zeros(
                        [g, out_shape_bnsd[-1]], dtype=torch.float32
                    )
                    continue
                q_curr = q_bnsd_tensor[batch, n2Idx * g : (n2Idx + 1) * g, s1Idx, :]
                sparse_indices_Indices = sparseIndicesIndices[batch, n2Idx, s1Idx, :]

                emptyFlag, k_sparse, v_sparse = gatherKV(
                    k_bnsd_tensor,
                    v_bnsd_tensor,
                    sparse_indices_Indices,
                    sparse_blocksize,
                    sparse_blockcount,
                    batch,
                    n2Idx,
                    s1Idx,
                    curr_actualSeq,
                    curr_actualSeq_q,
                    sparse_mode,
                )
                cur_sinks = None
                if sinks_tensor is not None:
                    cur_sinks = sinks_tensor[n2Idx * g : (n2Idx + 1) * g]
                if emptyFlag:
                    if cur_sinks is not None:
                        set_softmax_lse(
                            batch,
                            n2Idx,
                            s1Idx,
                            qLastPrefill,
                            cur_sinks,
                            torch.ones_like(cur_sinks),
                        )
                    continue
                bmm1Res = bmm1(
                    q_curr,
                    k_sparse,
                    torch.bfloat16
                    if fa_param["q_dtype"] == "bfloat16"
                    else torch.float16,
                )
                scaleRes = bmm1Res * scaleValue
                _, x_max, x_sum = softmax(scaleRes, cur_sinks)
                # 与算子softmax的数值链对齐: 未归一化 exp(x-max) 先量化为 P 的计算精度,bmm2 用 fp32 累加,最后除以fp32的sum
                exp_unnorm = torch.exp(scaleRes - x_max)
                if quantize_exp:
                    if fa_param["q_dtype"] == "float16":
                        exp_quantized = exp_unnorm.to(torch.float16).float()
                    elif fa_param["q_dtype"] == "bfloat16":
                        exp_quantized = exp_unnorm.to(torch.bfloat16).float()
                    else:
                        exp_quantized = exp_unnorm
                else:
                    # 三方标杆高精度路径: exp(x-max) 保持 fp32 直接参与 bmm2,
                    # 不量化到算子计算精度(对齐其他测试工具的高精度参考分支)
                    exp_quantized = exp_unnorm
                bmm2Res = torch.matmul(exp_quantized, v_sparse.float()) / x_sum
                y[batch, n2Idx * g : (n2Idx + 1) * g, s1Idx, :] = bmm2Res
                set_softmax_lse(
                    batch, n2Idx, s1Idx, qLastPrefill, x_max[:, 0], x_sum[:, 0]
                )
    return y, softmax_max, softmax_sum


def tensor_to_pa(
    tensor, block_table, blockTableShape, B, blockNum, blockSize, N, D, tensor_dtype
):
    tensor_cache = torch.zeros([blockNum, blockSize, N, D], dtype=tensor_dtype)
    if tensor.shape[1] != blockTableShape[1] * blockSize:
        tensor_old = tensor
        tensor = torch.zeros(
            (
                tensor_old.shape[0],
                blockTableShape[1] * blockSize,
                tensor_old.shape[2],
                tensor_old.shape[3],
            ),
            dtype=tensor_old.dtype,
        )
        tensor[:, : tensor_old.shape[1], :, :] = tensor_old

    for b in range(B):
        for block_i, kv_cache_blk_id in enumerate(block_table[b]):
            block_offset = block_i * blockSize
            if kv_cache_blk_id == -1:
                continue
            for n in range(N):
                tensor_cache[kv_cache_blk_id, 0:blockSize, n : n + 1, :] = tensor[
                    b, block_offset : (block_offset + blockSize), n : n + 1, :
                ]
    return tensor_cache


def kv_pa_preprocessing(input_tensor_dict, fa_param, params):
    N = fa_param["numKeyValueHeads"]
    actualSeqLengths_kv = fa_param["actualSeqLengths_kv"]
    blockSize = fa_param["blocksize"]
    block_table = fa_param["block_table"]
    layout_kv = fa_param["layout_kv"]

    k_dtype_str = params["dtype_input"].get("key", "bf16")
    _k_dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    k_dtype = _k_dtype_map.get(k_dtype_str)

    D = fa_param["k_shape_old"][-1]
    rope_head_dim = fa_param["k_rope_shape"][-1]

    B = len(fa_param["actualSeqLengths_kv"])

    k_tensor_bsnd = fa_param["_raw_key"].contiguous()
    k_rope_tensor_bsnd = fa_param["_raw_key_rope"].contiguous()
    k_rope_bsnd = (
        fa_param["_raw_key_rope"]
        if fa_param.get("_raw_key_rope") is not None
        else fa_param["k_rope_tensor"]
    )
    k_rope_bsnd = k_rope_bsnd.contiguous()
    blockTableShape = fa_param["bt_shape"]
    k_cache_base = tensor_to_pa(
        k_tensor_bsnd,
        block_table,
        blockTableShape,
        B,
        params["block_num"],
        blockSize,
        N,
        D,
        k_dtype,
    )
    k_cache_rope = tensor_to_pa(
        k_rope_tensor_bsnd,
        block_table,
        blockTableShape,
        B,
        params["block_num"],
        blockSize,
        N,
        rope_head_dim,
        k_dtype,
    )

    input_tensor_dict["key_cache"] = k_cache_base
    input_tensor_dict["value_cache"] = k_cache_base
    input_tensor_dict["key_rope_cache"] = k_cache_rope

    # kv_cache 0轴非连续
    key_stride = params.get("key_stride", 0)  # 0轴非连续增加stride
    if key_stride != 0:
        block_num = params["block_num"]
        base_kv = k_cache_base
        blocksize_with_stride = blockSize + key_stride
        blockFusion = torch.zeros(
            (block_num, blocksize_with_stride * N * D), dtype=k_dtype
        )
        base_flat = base_kv.reshape(block_num, blockSize * N * D)
        blockFusion[:, : blockSize * N * D] = base_flat
        base_kv_nc = blockFusion[:, : blockSize * N * D].view(
            block_num, blockSize, N, D
        )
        input_tensor_dict["key_cache"] = base_kv_nc
        input_tensor_dict["value_cache"] = base_kv_nc

        base_rope = k_cache_rope
        blockFusion_rope = torch.zeros(
            (block_num, blocksize_with_stride * N * rope_head_dim), dtype=k_dtype
        )
        rope_flat = base_rope.reshape(block_num, blockSize * N * rope_head_dim)
        blockFusion_rope[:, : blockSize * N * rope_head_dim] = rope_flat
        base_kvrope_nc = blockFusion_rope[:, : blockSize * N * rope_head_dim].view(
            block_num, blockSize, N, rope_head_dim
        )
        input_tensor_dict["key_rope_cache"] = base_kvrope_nc


def kv_nopa_preprocessing(input_tensor_dict, fa_param):
    input_tensor_dict["key_cache"] = fa_param["_raw_key"].contiguous()
    input_tensor_dict["value_cache"] = fa_param["_raw_key"].contiguous()
    input_tensor_dict["key_rope_cache"] = fa_param["_raw_key_rope"]


def qtensor_seqlength(q_shape, inputLayout):
    if inputLayout == "SH":
        return q_shape[0]
    elif inputLayout == "BSH":
        return q_shape[1]
    elif inputLayout == "NSD":
        return q_shape[1]
    elif inputLayout == "BSND":
        return q_shape[1]
    else:
        return q_shape[2]


def qtensor_dim(q_shape, layoutQuery, numhead):
    if layoutQuery == "SH":
        return q_shape[0]
    elif layoutQuery == "BSH" or layoutQuery == "BSH_NBSD":
        return int(q_shape[2] / numhead)
    else:
        return q_shape[-1]


def _save_test_case(input_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    case_name = input_data["Testcase_Name"]
    filepath = os.path.join(output_dir, f"{case_name}.pt")
    torch.save(input_data, filepath)
    print(f"测试用例已保存到: {filepath}")
    return filepath
